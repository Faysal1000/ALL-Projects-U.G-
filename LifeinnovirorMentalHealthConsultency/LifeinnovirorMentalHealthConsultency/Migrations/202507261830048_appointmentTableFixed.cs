namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class appointmentTableFixed : DbMigration
    {
        public override void Up()
        {
            AddColumn("dbo.Appointments", "FullName", c => c.String(nullable: false, maxLength: 100));
            AddColumn("dbo.Appointments", "Email", c => c.String(nullable: false));
            AddColumn("dbo.Appointments", "CancellationReason", c => c.String());
        }
        
        public override void Down()
        {
            DropColumn("dbo.Appointments", "CancellationReason");
            DropColumn("dbo.Appointments", "Email");
            DropColumn("dbo.Appointments", "FullName");
        }
    }
}
